# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Runtime evidence attached to computed results and retained with cached values.

The stamp itself is ``dynamix.core.execution``, re-exported here at its historical path; this
module adds the readings of a stamp and the recipe identity, which needs the engine's keys.
"""
from __future__ import annotations

from dynamix.core.execution import KEY, _plain, attach, metadata, outcome, run  # noqa: F401


def resolved_outcome(value, params):
    """The device's own resolved stamp, excluding copies of its declared parameters."""
    meta = metadata(value)
    record = meta.get('record') or {}
    result = {k: v for k, v in record.items() if k not in params or _plain(v) != _plain(params[k])}
    for key in ('_integration_order', '_integration_history'):
        if key in meta:
            result[key.removeprefix('_')] = meta[key]
    for key in ('preprocessing', 'fractional_integration'):
        if meta.get(key):
            last = meta[key][-1]
            result[key] = [{k: v for k, v in last.items()
                            if k not in params or _plain(v) != _plain(params[k])}]
    return _plain(result)


def recipe_identity(layer, layer_input, *, level=None, source=None):
    """Existing engine keys plus the applied filter parameters and shown level."""
    from dynamix.engine.resolve import source_identity, step_key
    from dynamix.model.contracts import step_states
    from dynamix.model.device import get_device, is_transform, keyed_params, validate_params

    key, filters = None, []
    sid = source_identity(layer, source)
    states = step_states(layer.chain.steps, layer_input, level_view=level is not None)
    for ref, state in zip(layer.chain.steps, states):
        if not state.applies:
            continue
        device = get_device(ref.device)
        params = validate_params(device, ref.params)
        if is_transform(device):
            key = step_key(device, sid, params, key)
        else:
            filters.append([device.name, _plain(keyed_params(device, params))])
    return {'analysis_key': key, 'filters': filters, 'level': level}
