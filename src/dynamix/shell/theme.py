# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Theme Rule: every color and font in the shell is a role resolved here.

A theme is DATA (Ableton's theme-file model): the default ships below, alternatives are new
Theme instances, and the QSS is generated -- no literal color or family name may appear in any
other shell module (tests/test_shell_boundaries.py enforces this).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

#: The bundled theme files (``<name>.json``) and their SVG assets (``assets/<name>/``).
THEMES_DIR = Path(__file__).resolve().parent / "themes"


@dataclasses.dataclass(frozen=True)
class Theme:
    #: The theme's name: its file stem under ``themes/`` and its assets folder.
    name: str = "restrained_dark"
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
    # The layer list's H/L/F/I toggles: black with white text at rest, inverted when set
    # (hidden / locked / frozen / inspector open), so the state reads at a glance.
    toggle_rest: str = "#000000"
    toggle_set: str = "#ffffff"
    sans_family: str = "Helvetica Neue"
    mono_family: str = "Menlo"
    base_pt: int = 11

    @classmethod
    def from_json(cls, path) -> "Theme":
        """A theme from its JSON file: exactly the keys of this dataclass (a missing or an unknown
        key is an error naming it), and the ink must read on the panel at 4.5:1 or better."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {f.name for f in dataclasses.fields(cls)}
        missing, unknown = sorted(fields - set(data)), sorted(set(data) - fields)
        if missing or unknown:
            raise ValueError(f"{Path(path).name}: missing {missing}, unknown {unknown}")
        theme = cls(**data)
        ratio = contrast(theme.ink, theme.panel)
        if ratio < 4.5:
            raise ValueError(f"{Path(path).name}: ink on panel reads at {ratio:.2f}:1, below 4.5:1")
        return theme


def _luminance(hex_color: str) -> float:
    """WCAG relative luminance of ``#rrggbb``."""
    digits = hex_color.lstrip("#")
    channels = [int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(hex_a: str, hex_b: str) -> float:
    """WCAG contrast ratio between two ``#rrggbb`` colours (1 to 21)."""
    la, lb = _luminance(hex_a), _luminance(hex_b)
    light, dark = max(la, lb), min(la, lb)
    return (light + 0.05) / (dark + 0.05)


def bundled_themes() -> dict[str, Path]:
    """Theme name -> its JSON file under ``themes/``."""
    return {p.stem: p for p in sorted(THEMES_DIR.glob("*.json"))}


def load_theme(name: str, user_dir=None) -> "Theme":
    """The bundled or user theme called ``name`` (a user file wins); KeyError when no file has it."""
    files = dict(bundled_themes())
    if user_dir is not None and Path(user_dir).is_dir():
        files.update({p.stem: p for p in sorted(Path(user_dir).glob("*.json"))})
    if name not in files:
        raise KeyError(f"no theme named {name!r}; known: {sorted(files)}")
    return Theme.from_json(files[name])


RESTRAINED_DARK = Theme()


def assets_dir(theme: Theme) -> Path:
    """The theme's SVG assets (``themes/assets/<name>/``), as the generated QSS references them."""
    return THEMES_DIR / "assets" / theme.name


def _url(theme: Theme, asset: str) -> str:
    return f'url("{(assets_dir(theme) / asset).as_posix()}")'


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
QToolButton[rowToggle="true"] {{ background: {t.toggle_rest}; color: {t.toggle_set};
                                border: 1px solid {t.border}; border-radius: 2px;
                                padding: 0px 4px; }}
QToolButton[rowToggle="true"]:checked {{ background: {t.toggle_set}; color: {t.toggle_rest}; }}
QListWidget {{ background: {t.panel}; border: none; }}
QListWidget::item:selected {{ background: {t.raised}; color: {t.amber}; }}
QLineEdit {{ background: {t.ground}; border: 1px solid {t.border};
             font-family: "{t.mono_family}"; }}
QLineEdit:focus {{ border: 1px solid {t.amber}; }}
QSlider::groove:horizontal {{ background: {t.raised}; height: 3px; }}
QSlider::handle:horizontal {{ background: {t.ink}; width: 8px; margin: -5px 0; }}
QSlider::handle:horizontal:focus {{ background: {t.amber}; }}
QCheckBox::indicator {{ width: 12px; height: 12px; image: {_url(t, "check_off.svg")}; }}
QCheckBox::indicator:checked {{ image: {_url(t, "check_on.svg")}; }}
QComboBox::down-arrow {{ image: {_url(t, "arrow_down.svg")}; width: 10px; height: 10px; }}
"""


def apply_theme(app, theme: Theme) -> None:
    from PySide6 import QtGui

    app.setStyleSheet(generate_qss(theme))
    app.setFont(QtGui.QFont(theme.sans_family, theme.base_pt))
