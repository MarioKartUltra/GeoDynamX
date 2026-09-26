# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Named starter chains for the browser's Racks section. Data only — no Qt."""
from __future__ import annotations

#: name -> ((device, params-overrides), ...). Params must validate against each
#: device's own schema; a test enforces it.
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
        ("scale_select", {"scale_idx": 0}),
        ("hline_length", {}),
    ),
    # Scrubbing IS scale_select, seeded in the default wtmm chain -- the multi-scale analyzers
    # get it from their rack instead of arriving bare and stacking every level.
    "CDF edges": (
        ("cdf_edges", {"n_levels": 4}),
        ("scale_select", {"scale_idx": 0}),
        ("hline_length", {}),
    ),
    # Perona-Malik proper: same multi-scale rack shape as CDF edges -- the
    # analyzer plus seeded scale scrubbing and orphan-confetti control.
    "PM edges": (
        ("pm_edges", {"n_levels": 4}),
        ("scale_select", {"scale_idx": 0}),
        ("hline_length", {}),
    ),
    # Single-scale output -- no scale_select needed; hline_length kills the orphan confetti.
    "Wavelet skeleton": (
        ("wavelet_skeleton", {}),
        ("hline_length", {"min_len": 10}),
    ),
}
