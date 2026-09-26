# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The right panel shows only the controls that apply to what is on screen."""
from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets  # noqa: F401

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from tests.test_shell_window import stub_devices, window  # noqa: F401


def _shown(win, key):
    panel = win.right_panel
    widget = panel._sections.get(key) or panel._relevant_items[key]
    return not widget.isHidden()


def _load(win, values, path="mem:r"):
    win._start_worker = lambda: None
    win.load_field(RasterField(name="r", values=values, frame=LocalFrame(),
                               x_axis=np.arange(values.shape[1], dtype=float),
                               y_axis=np.arange(values.shape[0], dtype=float)), path)


def _ext(n=5, lines=True, arg=True):
    e = {"x": np.arange(n), "y": np.arange(n), "mod": np.ones(n),
         "line_id": np.zeros(n, int) if lines else np.full(n, -1)}
    if arg:
        e["arg"] = np.zeros(n)
    return e


def test_a_raw_raster_shows_raster_controls_and_nothing_analysis_specific(window):
    win = window
    _load(win, np.random.default_rng(0).normal(size=(20, 24)))
    win._active_result = {}
    win._refresh_panel_relevance()
    for key in ("colormap", "stretch", "hillshade", "surface", "Transect", "Display"):
        assert _shown(win, key), key
    for key in ("swatches", "trails", "arrows", "knob:opacity", "knob:point_size",
                "knob:line_width", "reconstruct", "knob:sun_azimuth", "depth",
                "Skeleton", "Spectrum", "Decomposition", "Topology", "Groups",
                "Display mask — hides, never removes"):
        assert not _shown(win, key), key


def test_hillshade_and_surface_reveal_their_own_knobs(window):
    win = window
    _load(win, np.random.default_rng(0).normal(size=(20, 24)))
    win._on_display_style_changed("hillshade", True)
    assert _shown(win, "knob:sun_azimuth") and _shown(win, "knob:z_factor")
    win._on_display_style_changed("hillshade", False)
    assert not _shown(win, "knob:sun_azimuth") and not _shown(win, "knob:z_factor")
    win._on_display_style_changed("surface", True)
    assert _shown(win, "depth") and _shown(win, "knob:z_factor")


def test_each_tools_output_brings_exactly_its_controls(window):
    win = window
    _load(win, np.random.default_rng(0).normal(size=(20, 24)))
    # a WTMM result: maxima on lines, chains, tables
    win._active_result = {"extrema": [_ext()], "chains": [{"x": [1, 2], "y": [1, 2]}],
                          "hd_std": np.zeros(2), "hd_cmax": np.zeros(2)}
    win._refresh_panel_relevance()
    for key in ("swatches", "swatch:color_hchain", "swatch:color_vtrail", "trails", "arrows",
                "knob:opacity", "knob:line_width", "Skeleton", "Spectrum"):
        assert _shown(win, key), key
    assert not _shown(win, "reconstruct")
    # an edge detector's output: isolated maxima, no chains
    win._active_result = {"extrema": [_ext(lines=False, arg=False)]}
    win._refresh_panel_relevance()
    assert _shown(win, "swatch:color_extrema") and not _shown(win, "swatch:color_hchain")
    assert not _shown(win, "trails") and not _shown(win, "arrows")
    assert not _shown(win, "Skeleton")
    # a Hölder result: an h-map
    win._active_result = {"h_map": np.zeros((20, 24))}
    win._refresh_panel_relevance()
    assert _shown(win, "reconstruct") and _shown(win, "Spectrum")
    assert not _shown(win, "swatches")


def test_a_stack_hands_its_tone_to_the_composite(window):
    win = window
    _load(win, np.random.default_rng(0).normal(size=(12, 14, 3)), "mem:stack")
    win._active_result = {}
    win._refresh_panel_relevance()
    assert not _shown(win, "colormap") and not _shown(win, "stretch")
    assert not _shown(win, "hillshade") and _shown(win, "Composite")
