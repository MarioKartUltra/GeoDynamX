# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Device discovery: the user's devices folder and an installed package's entry points.

Registration stays explicit: an application calls :func:`discover` once, after the built-ins. A file
that fails to import or a device that fails to validate becomes a :class:`PluginResult` with the
reason and registers nothing; the next file is still tried. Every registered plugin records its
origin, the string the provenance writer and the cache salt use: ``user:<sha256 of the file>`` or
``package:<distribution>==<version>``; a built-in reads as ``builtin``."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

from dynamix.model.contracts import ALL_CONTRACTS, RECEIVED
from dynamix.model.device import register_device, validate_params

#: device id -> origin of a plugin device; a built-in is absent and reads as "builtin".
ORIGINS: dict[str, str] = {}

#: ``(user dir, entry group)`` -> the results of its first discovery. Discovery runs once per
#: process: a second window (the dev loop rebuilds one on every reload) gets the first call's
#: results and no file runs twice; an edited device file is picked up at the next start.
_DISCOVERED: dict[tuple[str | None, str], list[PluginResult]] = {}


@dataclasses.dataclass(frozen=True)
class PluginResult:
    name: str                  # the device id, or "" when the file never produced one
    origin: str                # "user:<sha256>" or "package:<dist>==<version>"
    error: str | None = None   # None when registered
    path: str | None = None    # the file, for a user device


def origin_of(name: str) -> str:
    return ORIGINS.get(name, "builtin")


def validate_plugin(device) -> None:
    """What :func:`register_device` does not check: the contract names, the declared capabilities,
    the role declarations and a working default set."""
    from dynamix.model.capabilities import declared
    from dynamix.model.roles import band_for_analyzer

    names: set[str] = set()
    for alternative in getattr(device, "consumes", ()):
        names.update(alternative)
    produces = getattr(device, "produces", None)
    if produces is not None:
        names.update(produces)
    unknown = sorted(n for n in names if n != RECEIVED and n not in ALL_CONTRACTS)
    if unknown:
        raise ValueError(f"unknown contract name(s) {unknown}; known: {sorted(ALL_CONTRACTS)}")
    declared(device)                            # raises on an unknown capability name
    role = getattr(device, "role", None)
    if role not in (None, "primary"):
        raise ValueError(f"role must be absent or 'primary', not {role!r}")
    for attr in ("superseded_by", "band_variant_of"):
        value = getattr(device, attr, None)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{attr} must be a device id (a string), not {value!r}")
    variant_of = getattr(device, "band_variant_of", None)
    if variant_of:
        rank = getattr(device, "band_rank", None)
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(f"band_variant_of = {variant_of!r} needs an integer band_rank")
        if variant_of in band_for_analyzer():
            raise ValueError(f"{variant_of!r} already has a band variant, "
                             f"{band_for_analyzer()[variant_of]!r}; a plugin does not replace it")
    validate_params(device, {})


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _register(device, origin: str, path: str | None) -> PluginResult:
    name = getattr(device, "name", "")
    try:
        validate_plugin(device)
        register_device(device)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:    # somebody else's code: any failure (a sys.exit too) is a row
        return PluginResult(name, origin, _failure(exc), path)
    ORIGINS[name] = origin
    return PluginResult(name, origin, None, path)


def _load_file(path: Path) -> list[PluginResult]:
    origin = f"user:{path.name}"            # until the file is read: an unreadable file is a row too
    try:
        origin = f"user:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        spec = importlib.util.spec_from_file_location(f"geodynamix_user_device_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        devices = list(getattr(module, "DEVICES"))
    except KeyboardInterrupt:
        raise
    except BaseException as exc:    # a script that ends in sys.exit() is a row, never the app's exit
        return [PluginResult("", origin, _failure(exc), str(path))]
    return [_register(d, origin, str(path)) for d in devices]


def discover(user_dir: Path | None, entry_group: str = "geodynamix.devices") -> list[PluginResult]:
    """Register every device under ``user_dir/*.py`` (each module exposes ``DEVICES``) and every
    ``entry_group`` entry point (its value a module attribute holding ``DEVICES``); one result per
    device, or per file or entry point that produced none. A repeat call for the same folder and
    group returns the first call's results."""
    memo = (None if user_dir is None else str(user_dir), entry_group)
    if memo in _DISCOVERED:
        return list(_DISCOVERED[memo])
    results: list[PluginResult] = []
    if user_dir is not None and Path(user_dir).is_dir():
        for path in sorted(Path(user_dir).glob("*.py")):
            if path.name.startswith("test_") or path.name.startswith("_"):
                continue
            results.extend(_load_file(path))
    for ep in importlib.metadata.entry_points(group=entry_group):
        dist = getattr(ep, "dist", None)
        origin = (f"package:{dist.name}=={dist.version}" if dist is not None
                  else f"package:{ep.value}")
        try:
            devices = list(ep.load())
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            results.append(PluginResult("", origin, _failure(exc), None))
            continue
        results.extend(_register(d, origin, None) for d in devices)
    _DISCOVERED[memo] = results
    return list(results)
