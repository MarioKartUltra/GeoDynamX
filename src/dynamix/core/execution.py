# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Runtime evidence attached to computed results and retained with cached values.

It lives at the bottom layer so the ROI runner, below the engine, stamps what it computes the
same way the engine does; ``dynamix.engine.execution`` re-exports every name here.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.metadata
import platform

from dynamix.core import fft_policy

KEY = '_execution'


def metadata(value):
    return value if isinstance(value, dict) else (getattr(value, 'provenance', None) or {})


def attach(value, **entries):
    if tuple(entries) == ('_landed',) and metadata(value).get('_landed') == entries['_landed']:
        return value
    if isinstance(value, dict):
        return {**value, **entries}
    if dataclasses.is_dataclass(value) and hasattr(value, 'provenance'):
        return dataclasses.replace(value, provenance={**value.provenance, **entries})
    return value


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return _plain(value.tolist()) if hasattr(value, 'tolist') else value


def outcome(device, params, value):
    hook = getattr(device, 'provenance_outcome', None)
    if hook is not None:
        return _plain(hook(value, params))
    return {}


def run(device, params, callback, source=None, *, represented=()):
    """Capture effective numerical context inside the call's own precision scope."""
    with fft_policy.scope_for(device, params), fft_policy.observing() as calls:
        fft_policy.include_observed(represented)
        value = callback()
    # These are dependencies of the method, not every library another open tool loaded.
    libraries = {'python': platform.python_version()}
    for module, distribution in [('numpy', 'numpy'), *getattr(device, 'provenance_libraries', ())]:
        try:
            libraries[module] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            libraries[module] = None
    context = {'libraries': libraries, 'platform': f'{platform.platform(terse=True)}-{platform.machine()}'}
    if len(calls) == 1:
        context.update(copy.deepcopy(calls[0]))
    elif not calls and getattr(device, 'provenance_precision', None):
        context['precision'] = device.provenance_precision
    own = {'device': device.name, 'context': context, 'fft': copy.deepcopy(calls),
           'outcome': outcome(device, params, value)}
    steps = copy.deepcopy(metadata(source).get(KEY, []))
    return attach(value, **{KEY: [*steps, own]})
