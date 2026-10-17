# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Where a device's code comes from: its authors, licence and citation.

A device may declare ``origin = Origin(...)``; one that declares nothing is first-party
(:data:`FIRST_PARTY`). The browser shows it in the tooltip, the device reference lists it and the
``.gprv`` writer records it on every step. This is the who-wrote-it axis; the run-time axis (a user
file, an installed package) is ``dynamix.model.plugins.origin_of``."""
from __future__ import annotations

import dataclasses

#: SPDX ids GPL-2.0-or-later code may combine with (THIRD_PARTY.md).
ACCEPTED_LICENSES: frozenset[str] = frozenset({
    "MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "LGPL-2.1-or-later", "LGPL-3.0-or-later",
    "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-or-later",
})

#: ``kind`` values: written here; translated or ported from another program; an external tree
#: under ``third_party/`` reached through an adapter.
KINDS: tuple[str, ...] = ("first_party", "derived", "third_party")

AUTHOR = "Abraham Joseph Okayli Masaryk"
REPOSITORY = "https://github.com/MarioKartUltra/GeoDynamX"


@dataclasses.dataclass(frozen=True)
class Origin:
    kind: str
    authors: tuple[str, ...]
    license: str
    citation: str = ""
    url: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"origin kind {self.kind!r}; known: {KINDS}")
        if self.license not in ACCEPTED_LICENSES:
            raise ValueError(f"licence {self.license!r} is not accepted; see THIRD_PARTY.md")
        if not self.authors:
            raise ValueError("an origin names at least one author")

    def lines(self) -> list[str]:
        """The tooltip and reference lines: authors, licence (with the kind when not first-party),
        citation, URL; empty fields are left out."""
        kind = {"derived": " (derived)", "third_party": " (third party)"}.get(self.kind, "")
        out = [", ".join(self.authors), f"{self.license}{kind}"]
        if self.citation:
            out.append(self.citation)
        if self.url:
            out.append(self.url)
        return out


FIRST_PARTY = Origin("first_party", (AUTHOR,), "GPL-2.0-or-later", url=REPOSITORY)

#: The WTMM chaining and follow code is a line-for-line port of xsmurf (NOTICE), so a device that
#: runs it is GPL-2.0-only.
XSMURF_DERIVED = Origin(
    "derived", (AUTHOR, "Nicolas Decoster", "Pierre Kestener", "Stéphane Roux", "Alain Arneodo"),
    "GPL-2.0-only",
    citation="Arneodo, Decoster & Roux (2000), Eur. Phys. J. B 15, 567; chaining and follow ported "
             "from xsmurf, Centre de Recherche Paul Pascal, Bordeaux (1999)",
    url=REPOSITORY)

#: The Mallat–Zhong edges and their reconstruction are a port of LastWave's dwtrans2d (NOTICE).
LASTWAVE_DERIVED = Origin(
    "derived", (AUTHOR, "Emmanuel Bacry", "Jérôme Fraleu", "Jérôme Kalifa", "Erwan Le Pennec",
                "Wen-Liang Hwang", "Stéphane Mallat", "Sifen Zhong"),
    "GPL-2.0-or-later",
    citation="Mallat & Zhong (1992), IEEE Trans. Pattern Anal. Mach. Intell. 14(7), 710; ported "
             "from LastWave's dwtrans2d 2.1 (1998–2002)",
    url=REPOSITORY)


def origin_of_device(device) -> Origin:
    """The device's declared origin, or :data:`FIRST_PARTY`."""
    declared = getattr(device, "origin", None)
    if declared is None:
        return FIRST_PARTY
    if not isinstance(declared, Origin):
        raise TypeError(f"{getattr(device, 'name', device)!r}: origin must be an Origin, not {type(declared).__name__}")
    return declared
