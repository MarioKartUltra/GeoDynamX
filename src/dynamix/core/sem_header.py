# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Microscope image headers: the pixel size an SEM wrote beside its image.

No cross-vendor standard carries an SEM's pixel size. The TIFF resolution tags hold a print
density, OME-TIFF's PhysicalSizeX is the one open standard and the SEM vendors do not write
it, and each vendor keeps its own block: TESCAN a sidecar ``<name>-<ext>.hdr`` in INI form
(``[MAIN] PixelSizeX`` in metres, ``ImageStripSize`` rows of info strip under the image),
Thermo/FEI an INI block in TIFF tag 34682, Zeiss text in tag 34118, JEOL and Hitachi text
sidecars. One reader per vendor fills the same :class:`~dynamix.core.frames.LocalFrame`;
TESCAN is the first.
"""
from __future__ import annotations

import configparser
from pathlib import Path

#: Suffixes an image beside a bare ``.hdr`` may carry, ENVI's binaries last (``""``: no suffix).
_IMAGE_SUFFIXES = (".tif", ".tiff", ".img", ".dat", ".bsq", ".bil", ".bip", ".raw", "")


def _first_line(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                return line.strip()
    return ""


def tescan_sidecar(image) -> "Path | None":
    """TESCAN's header beside ``image``: ``<stem>-<ext>.hdr`` (``map.tif`` -> ``map-tif.hdr``),
    else ``<stem>.hdr`` when its first line opens the ``[MAIN]`` section."""
    image = Path(image)
    named = image.with_name(f"{image.stem}-{image.suffix.lstrip('.')}.hdr")
    if named.is_file():
        return named
    bare = image.with_suffix(".hdr")
    if bare.is_file() and _first_line(bare).upper() == "[MAIN]":
        return bare
    return None


def read_tescan(hdr) -> dict:
    """The header's sections: ``{"vendor", "file", "pixel_size_m": (x, y), "strip_rows",
    "fields": {section: {key: value}}}``, the pixel sizes in metres as the file states them."""
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read_string(Path(hdr).read_text(encoding="utf-8", errors="replace"))
    main = parser["MAIN"]
    px = float(main["PixelSizeX"])
    py = float(main.get("PixelSizeY", main["PixelSizeX"]))
    strip = int(float(main["ImageStripSize"])) if "ImageStripSize" in main else 0
    return {"vendor": "TESCAN", "file": str(hdr), "pixel_size_m": (px, py), "strip_rows": strip,
            "fields": {s: dict(parser[s]) for s in parser.sections()}}


def read_sidecar(image) -> "dict | None":
    """The vendor header beside ``image``, parsed, or ``None``."""
    hdr = tescan_sidecar(image)
    return read_tescan(hdr) if hdr is not None else None


def partner_of(hdr) -> "Path | None":
    """The image a ``.hdr`` describes: TESCAN's ``<stem>-<ext>.hdr`` names its image's suffix;
    any other header (ENVI's, a bare TESCAN one) sits beside its image under the same stem.
    ``None`` when nothing beside it matches."""
    hdr = Path(hdr)
    stem, dash, ext = hdr.stem.rpartition("-")
    if dash and _first_line(hdr).upper() == "[MAIN]":
        image = hdr.with_name(f"{stem}.{ext}")
        if image.is_file():
            return image
    for suffix in _IMAGE_SUFFIXES:
        candidate = hdr.with_suffix(suffix)
        if candidate != hdr and candidate.is_file():
            return candidate
    return None
