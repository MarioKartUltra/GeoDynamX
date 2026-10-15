# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The device sets the shell used to keep by hand, derived from declarations on the devices.

A device declares ``role = "primary"`` (an analyzer a drop forks a sibling layer for),
``chain_producer = True`` (its extrema chains feed the selection filters), ``chain_consumer = True``
(a filter that consumes those chains), ``superseded_by = "<id>"`` (a successor the palette shows
instead; ``""`` when the Transport took over), ``shell_placed = True`` (only the shell places it),
``band_variant_of = "<analyzer>"`` with ``band_rank`` (a band reconstruction committed for that
analyzer's band fork; rank 0 newest). Every function reads the registry as it stands, so a device
dropped in the user's folder joins the sets by its own declarations."""
from __future__ import annotations

from dynamix.model.device import DEVICES


def _with(attr: str, value=True) -> tuple[str, ...]:
    return tuple(name for name, d in DEVICES.items() if getattr(d, attr, None) == value)


def primary_analyzers() -> tuple[str, ...]:
    return _with("role", "primary")


def chain_producers() -> tuple[str, ...]:
    return _with("chain_producer")


def consumer_producers() -> dict[str, tuple[str, ...]]:
    """Each chain-consuming filter -> the producers whose chains it reads."""
    producers = chain_producers()
    return {name: producers for name in _with("chain_consumer")}


def superseded() -> dict[str, str]:
    """Superseded device -> its successor (``""`` when a shell control took the job)."""
    return {name: d.superseded_by for name, d in DEVICES.items()
            if getattr(d, "superseded_by", None) is not None}


def shell_placed() -> tuple[str, ...]:
    return _with("shell_placed")


def band_for_analyzer() -> dict[str, str]:
    """Analyzer -> the band-reconstruction variant its band fork commits."""
    variants = sorted(((d.band_rank, d.band_variant_of, name) for name, d in DEVICES.items()
                       if getattr(d, "band_variant_of", None)), key=lambda t: t[0])
    return {analyzer: name for _rank, analyzer, name in variants}


def band_devices() -> tuple[str, ...]:
    """The band-reconstruction variants, newest first."""
    return tuple(band_for_analyzer().values())


def is_dev(name: str, device) -> bool:
    """A stub, a superseded or shell-placed device, or one that declares an empty families
    tuple: listed under the browser's folded "Dev" heading."""
    return (name.startswith("stub_") or getattr(device, "superseded_by", None) is not None
            or bool(getattr(device, "shell_placed", False))
            or getattr(device, "families", None) == ())
