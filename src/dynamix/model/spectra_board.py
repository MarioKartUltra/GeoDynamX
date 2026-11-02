# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The D(h) comparison board: the spectra sent from the fitting windows, in legend order.

Session state the main window holds; nothing here is written to the project file. An entry is a
snapshot of its producer's spectrum when it was sent. A send adds an entry, two scale windows of
one method two entries side by side; a send exactly identical to an entry (one kind, one
identity: the producing result's tool identity, the scale window, the fit mode, the table and
the producer's settings) updates that entry in place instead, keeping its display state (its
display name when renamed, its colour, its legend position, its visibility, its bars and tangent
switches). A replace swaps the most recently sent entry of one producer (the window and the
producing tool, without the scale window) and kind for a new fit, keeping the same display
state. ``name`` is the
display name and the only name a rename touches; ``default_name`` is the display name the
producer gave, which an empty rename restores; ``full_name`` is the producer's own name, with
the scale window and the tangent it was sent under."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

import numpy as np

__all__ = ["KINDS", "SentSpectrum", "SpectraBoard", "plain"]

#: What produced a spectrum: the partition fitter's canonical fit, its histogram and box-counting
#: constructions, the generalized (Touchette-Beck) inversion, large deviations, the log-cumulant
#: expansion, the structure functions, and the partition fitter over the DWT or CWT leaders' table.
KINDS = ("canonical", "histogram", "boxcount", "generalized", "large_deviations", "cumulant",
         "structure_functions", "dwt_leaders", "cwt_leaders")

#: The keys a carried tangent holds: the point of contact, its slope (q_c) and its D-intercept.
TANGENT_KEYS = ("q_c", "h_c", "D_c", "slope", "intercept")


def _array(values) -> "np.ndarray | None":
    return None if values is None else np.asarray(values, dtype=np.float64).ravel()


def plain(value):
    """``value`` in JSON's types: arrays and tuples as lists, numpy scalars as Python numbers,
    non-finite floats as None."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


@dataclass(eq=False)
class SentSpectrum:
    """One spectrum on the board: ``h`` and ``D`` with their error bars (None without), the
    partition-side ``q`` and ``tau`` when the producer has them, the phase-transition tangent
    when it was sent with one, the identity an exactly identical send matches, the producer (the
    sending window and the producing tool, without the scale window) a replace matches, and the
    display state: ``renamed`` once a rename gave it a name of its own."""

    name: str
    full_name: str
    kind: str
    h: np.ndarray
    D: np.ndarray
    dh: "np.ndarray | None" = None
    dD: "np.ndarray | None" = None
    q: "np.ndarray | None" = None
    tau: "np.ndarray | None" = None
    tangent: "dict | None" = None
    identity: dict = field(default_factory=dict)
    producer: dict = field(default_factory=dict)
    colour: str = ""
    visible: bool = True
    show_errors: bool = True
    show_tangent: bool = True
    default_name: str = ""
    renamed: bool = False
    key: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown spectrum kind {self.kind!r}; expected one of {KINDS}")
        self.h, self.D = _array(self.h), _array(self.D)
        if self.h.shape != self.D.shape:
            raise ValueError(f"h and D differ in length: {self.h.size} and {self.D.size}")
        self.dh, self.dD, self.q, self.tau = (_array(a) for a in (self.dh, self.dD, self.q, self.tau))
        for name, err in (("dh", self.dh), ("dD", self.dD)):
            if err is not None and err.shape != self.h.shape:
                raise ValueError(f"{name} differs in length from h: {err.size} and {self.h.size}")
        if (self.q is None) != (self.tau is None) or (self.q is not None and self.q.shape != self.tau.shape):
            raise ValueError("q and tau come together, of one length")
        if self.tangent is not None:
            missing = [k for k in TANGENT_KEYS if k not in self.tangent]
            if missing:
                raise ValueError(f"the tangent lacks {missing}")
            self.tangent = {k: float(self.tangent[k]) for k in TANGENT_KEYS}
        self.name = str(self.name or self.full_name)
        self.default_name = str(self.default_name or self.name)

    @property
    def has_errors(self) -> bool:
        return self.dD is not None or self.dh is not None

    def same_as(self, other: "SentSpectrum") -> bool:
        """Whether ``other`` is this spectrum sent again: one kind, one identity."""
        return self.kind == other.kind and _identity_text(self.identity) == _identity_text(other.identity)

    def record(self) -> dict:
        """The entry as a figure record lists it: the names, the kind, the identity, the scale
        window, the fit mode and the table, whether its error bars were drawn and the tangent
        drawn with it."""
        ident = dict(self.identity)
        return plain({"name": self.name, "full_name": self.full_name, "kind": self.kind,
                      "identity": {k: v for k, v in ident.items()
                                   if k not in ("scale_window", "fit_mode", "table")},
                      "scale_window": ident.get("scale_window"), "fit_mode": ident.get("fit_mode"),
                      "table": ident.get("table"),
                      "errors": bool(self.show_errors and self.has_errors),
                      "tangent": self.tangent if self.show_tangent else None,
                      "colour": self.colour})


def _identity_text(identity: dict) -> str:
    return json.dumps(plain(identity), sort_keys=True)


class SpectraBoard:
    """The sent spectra in board order (the legend's order)."""

    def __init__(self) -> None:
        self._entries: list[SentSpectrum] = []
        self._sent: dict[str, int] = {}             # key -> when last sent or replaced
        self._clock = 0

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[SentSpectrum]:
        return list(self._entries)

    def get(self, key: str) -> SentSpectrum:
        for entry in self._entries:
            if entry.key == key:
                return entry
        raise KeyError(key)

    def _stamp(self, key: str) -> str:
        self._clock += 1
        self._sent[key] = self._clock
        return key

    def _take_over(self, i: int, spectrum: SentSpectrum) -> str:
        """``spectrum`` in place of entry ``i``: the data and the producer's display name are the
        new ones; the key, the legend position, the name a rename gave, the colour, the
        visibility and the bars and tangent switches the old entry's."""
        old = self._entries[i]
        spectrum.key, spectrum.colour, spectrum.visible = old.key, old.colour, old.visible
        spectrum.show_errors, spectrum.show_tangent = old.show_errors, old.show_tangent
        if old.renamed:
            spectrum.name, spectrum.renamed = old.name, True
        self._entries[i] = spectrum
        return self._stamp(spectrum.key)

    def add(self, spectrum: SentSpectrum) -> str:
        """Append ``spectrum``, or update in place the entry it is exactly identical to (one
        kind, one identity). Returns the entry's key."""
        for i, old in enumerate(self._entries):
            if old.same_as(spectrum):
                return self._take_over(i, spectrum)
        self._entries.append(spectrum)
        return self._stamp(spectrum.key)

    def replace(self, key: str, spectrum: SentSpectrum) -> str:
        """``spectrum`` in place of the entry ``key``, keeping its key, legend position, colour,
        visibility, switches and a name a rename gave. Returns the key."""
        return self._take_over(self._entries.index(self.get(key)), spectrum)

    def latest_of(self, kind: str, producer: dict) -> "str | None":
        """The key of the most recently sent or replaced entry of ``kind`` from ``producer``,
        else None."""
        text = _identity_text(producer)
        keys = [e.key for e in self._entries if e.kind == kind and _identity_text(e.producer) == text]
        return max(keys, key=self._sent.__getitem__, default=None)

    def remove(self, key: str) -> None:
        self._entries.remove(self.get(key))
        self._sent.pop(key, None)

    def rename(self, key: str, name: str) -> None:
        """The display name; an empty one restores the producer's display name."""
        entry, text = self.get(key), str(name).strip()
        entry.name = text or entry.default_name
        entry.renamed = entry.name != entry.default_name

    def set_visible(self, key: str, on: bool) -> None:
        self.get(key).visible = bool(on)

    def set_show_errors(self, key: str, on: bool) -> None:
        self.get(key).show_errors = bool(on)

    def set_show_tangent(self, key: str, on: bool) -> None:
        self.get(key).show_tangent = bool(on)

    def set_colour(self, key: str, colour: str) -> None:
        self.get(key).colour = str(colour)

    def move(self, key: str, step: int) -> None:
        """Move the entry ``step`` places (negative: towards the front), stopping at the ends."""
        entry = self.get(key)
        i = self._entries.index(entry)
        j = min(max(i + int(step), 0), len(self._entries) - 1)
        self._entries.insert(j, self._entries.pop(i))
