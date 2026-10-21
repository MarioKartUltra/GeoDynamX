# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""core.execution stamps the runtime evidence the ROI runner attaches; engine.execution re-exports
it (the layering gate), so a result stamped below the engine carries the record the export reads."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import make_field
from dynamix.core import fft_policy

MODULES = ["dynamix.core.execution", "dynamix.engine.execution"]


@pytest.mark.parametrize("module", MODULES)
def test_run_appends_its_step_to_the_source_record(module):
    execution = importlib.import_module(module)
    fft_policy.configure("fftw", 64)
    device = SimpleNamespace(name="probe", params=(SimpleNamespace(name="precision"),),
                             provenance_outcome=lambda value, params: {"peaks": np.arange(2)})
    source = {execution.KEY: [{"device": "upstream"}]}
    value = execution.run(device, {"precision": "f32"},
                          lambda: {"raster": fft_policy.active().fft2(np.ones((4, 4))).real}, source)
    upstream, own = value["_execution"]
    assert upstream == {"device": "upstream"}
    assert set(own) == {"device", "context", "fft", "outcome"}
    assert own["device"] == "probe" and own["outcome"] == {"peaks": [0, 1]}
    assert set(own["context"]) == {"libraries", "platform", "backend", "precision"}
    assert own["context"]["precision"] == "float32"
    assert [call["precision"] for call in own["fft"]] == ["float32"]
    assert own["context"]["backend"] == own["fft"][0]["backend"]


@pytest.mark.parametrize("module", MODULES)
def test_a_field_carries_its_record_in_its_provenance(module):
    execution = importlib.import_module(module)
    field = make_field(shape=(8, 8))
    stamped = execution.run(SimpleNamespace(name="stage", provenance_precision="float64"), {},
                            lambda: field)
    [own] = execution.metadata(stamped)["_execution"]
    assert own["device"] == "stage" and own["fft"] == [] and own["outcome"] == {}
    assert own["context"]["precision"] == "float64"
    assert execution.attach(stamped, extra=1).provenance["extra"] == 1
    assert "_execution" not in field.provenance
