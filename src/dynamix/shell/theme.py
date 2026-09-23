# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Theme Rule: every color and font in the shell is a role resolved here.

A theme is DATA (Ableton's theme-file model): the default ships below, alternatives are new
Theme instances, and the QSS is generated -- no literal color or family name may appear in any
other shell module (tests/test_shell_boundaries.py enforces this).
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Theme:
    # tonal ramp, dark -> light (The Instrument Rack: ground -> zone -> panel -> raised)
    ground: str = "#131313"     # canvas surround
    zone: str = "#191919"       # chain-strip zone (its own neutral, per DESIGN.md)
    panel: str = "#1d1d1d"      # side panel / strips at rest
    raised: str = "#272727"     # hover / raised strip
    border: str = "#303030"
    ink: str = "#e6e6e6"        # labels, readings
    ink_muted: str = "#a0a0a0"  # secondary labels (still >= 4.5:1 on panel)
    amber: str = "#ffa028"      # selection / active / focus ONLY (One-Accent Rule)
    error: str = "#ff5c5c"
    seam: str = "#ff7043"       # seam-flag warning (canvas.SEAM_COLOR matches this value exactly)
    # The CHAIN-selection accent -- box rubber-band,
    # lasso capture band, and the visible selected-chain overlay (Canvas.set_selection_chains) all
    # share this ONE role. Deliberately its OWN field rather than a reuse of `amber`: both sit in
    # the same amber/yellow family and both mean "this is what the user is selecting right now"
    # under the One-Accent Rule, but `amber` is spent narrowly elsewhere (the ⌘-drag ROI band,
    # focus rings) and a future re-tint of one must not silently re-tint the other (the same
    # "own constant, pinned relationship" doctrine `tests/test_shell_boundaries.py` already
    # enforces for VTRAIL_COLOR/ROI_BOUNDS_COLOR/SEAM_COLOR against their own theme roles).
    # Chosen as a value BETWEEN EQSelect's own selection yellow (`_SELECT_RGB`, hex ffeb3b) and this theme's more
    # orange `amber` (hex ffa028): close enough to read as the same "selection" family, distinct
    # enough that the two roles can diverge later without a hidden coupling.
    selection_accent: str = "#ffc107"
    sans_family: str = "Helvetica Neue"
    mono_family: str = "Menlo"
    base_pt: int = 11


RESTRAINED_DARK = Theme()


def generate_qss(theme: Theme) -> str:
    """Application stylesheet from role tokens. Flat: no bevels, no gradients, no radii > 2px.

    The ``state`` selector family colours a transform strip's state dot (chain_strip.py sets the
    property; only the QSS knows what a state looks like). ``computing`` is the one that gets the
    accent -- it is the only transient state, and the One-Accent Rule spends amber on what is
    happening NOW. ``idle`` and ``cached`` are both muted on purpose: a cached frame is the resting
    state of this app, not an event.
    """
    t = theme
    return f"""
QWidget {{ background: {t.panel}; color: {t.ink};
           font-family: "{t.sans_family}"; font-size: {t.base_pt}pt; }}
QMainWindow, QGraphicsView {{ background: {t.ground}; }}
QLabel[reading="true"] {{ font-family: "{t.mono_family}"; color: {t.ink}; }}
QLabel[muted="true"] {{ color: {t.ink_muted}; }}
QLabel[state="idle"] {{ color: {t.ink_muted}; }}
QLabel[state="computing"] {{ color: {t.amber}; }}
QLabel[state="pending"] {{ color: {t.amber}; }}
QLabel[state="cached"] {{ color: {t.ink_muted}; }}
QLabel[state="error"] {{ color: {t.error}; }}
QFrame[zone="strip"] {{ background: {t.zone}; }}
QFrame[zone="strip"][refused="true"] {{ background: {t.error}; }}
QFrame[strip="true"] {{ background: {t.panel}; border: 1px solid {t.border}; border-radius: 2px; }}
QFrame[strip="true"][selected="true"] {{ border: 1px solid {t.amber}; }}
QFrame[strip="true"]:hover {{ background: {t.raised}; }}
QPushButton {{ background: {t.raised}; border: 1px solid {t.border}; border-radius: 2px;
               padding: 2px 8px; }}
QPushButton:focus {{ border: 1px solid {t.amber}; outline: none; }}
QPushButton:checked {{ color: {t.amber}; }}
QListWidget {{ background: {t.panel}; border: none; }}
QListWidget::item:selected {{ background: {t.raised}; color: {t.amber}; }}
QLineEdit {{ background: {t.ground}; border: 1px solid {t.border};
             font-family: "{t.mono_family}"; }}
QLineEdit:focus {{ border: 1px solid {t.amber}; }}
QSlider::groove:horizontal {{ background: {t.raised}; height: 3px; }}
QSlider::handle:horizontal {{ background: {t.ink}; width: 8px; margin: -5px 0; }}
QSlider::handle:horizontal:focus {{ background: {t.amber}; }}
"""


def apply_theme(app, theme: Theme) -> None:
    from PySide6 import QtGui

    app.setStyleSheet(generate_qss(theme))
    app.setFont(QtGui.QFont(theme.sans_family, theme.base_pt))
