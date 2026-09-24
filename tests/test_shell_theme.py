# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Theme roles and QSS generation. DESIGN.md's Theme Rule made testable."""
import re

from dynamix.shell.theme import RESTRAINED_DARK, Theme, generate_qss


def _rel_lum(hexcolor: str) -> float:
    r, g, b = (int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5))
    def f(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_rel_lum(a), _rel_lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_default_theme_roles_exist():
    t = RESTRAINED_DARK
    for role in ("ground", "panel", "raised", "zone", "border",
                 "ink", "ink_muted", "amber", "error", "seam",
                 "sans_family", "mono_family", "base_pt"):
        assert getattr(t, role)


def test_seam_role_is_distinct_from_amber_and_error():
    """One-Accent Rule guard: the seam-flag role must not collide with the amber accent
    (selection/active/focus ONLY) or the error role -- three different meanings need three
    different colors, or a flagged chain would read as a selection or a fault."""
    t = RESTRAINED_DARK
    assert t.seam != t.amber
    assert t.seam != t.error


def test_contrast_commitments():
    t = RESTRAINED_DARK
    assert _contrast(t.ink, t.panel) >= 4.5
    assert _contrast(t.ink_muted, t.panel) >= 4.5     # muted may not fail WCAG
    assert _contrast(t.amber, t.panel) >= 3.0          # large/bold accent floor


def test_tonal_ramp_orders_dark_to_light():
    t = RESTRAINED_DARK
    lums = [_rel_lum(c) for c in (t.ground, t.zone, t.panel, t.raised)]
    assert lums == sorted(lums)


def test_qss_generation_is_pure_and_complete():
    qss = generate_qss(RESTRAINED_DARK)
    assert RESTRAINED_DARK.panel in qss and RESTRAINED_DARK.amber in qss
    assert "{" in qss and qss.count("{") == qss.count("}")
    assert len(re.findall(r"#[0-9a-fA-F]{6}\b", qss)) == qss.count("#")  # every # is a 6-hex color


def test_qss_carries_the_state_dot_family():
    """A transform strip's state dot is coloured entirely by QSS (chain_strip.py only sets the
    property), so a missing selector means an invisible state, not a styling nit."""
    qss = generate_qss(RESTRAINED_DARK)
    for state in ("idle", "computing", "cached", "error"):
        assert f'QLabel[state="{state}"]' in qss
    assert RESTRAINED_DARK.error in qss


def test_theme_is_frozen_data():
    import dataclasses
    assert dataclasses.is_dataclass(RESTRAINED_DARK)
    try:
        RESTRAINED_DARK.amber = "#ff0000"
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised


def test_row_toggles_read_black_at_rest_and_white_when_set():
    """The layer list's H/L/F/I toggles: black with white text at rest, white with black text
    when set (hidden / locked / frozen / inspector open) -- styled here, not left to the
    platform, whose checked state was hard to tell from unchecked."""
    t = RESTRAINED_DARK
    qss = generate_qss(t)
    rest = re.search(r'QToolButton\[rowToggle="true"\] \{([^}]*)\}', qss)
    checked = re.search(r'QToolButton\[rowToggle="true"\]:checked \{([^}]*)\}', qss)
    assert rest and checked
    assert f"background: {t.toggle_rest};" in rest.group(1)
    assert f"color: {t.toggle_set};" in rest.group(1)
    assert f"background: {t.toggle_set};" in checked.group(1)
    assert f"color: {t.toggle_rest};" in checked.group(1)
    assert (t.toggle_rest, t.toggle_set) == ("#000000", "#ffffff")
