# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import pytest
from PySide6 import QtCore

from dynamix.engine import Cache
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.layer import Layer


class _SlowStub:
    name = "slow_stub"
    params = ()
    def compute(self, field, params, *, progress=None):
        if progress:
            progress("stub", 0.0); progress("stub", 1.0)
        return {"ok": True, "field_sum": float(field.sum())}
    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


@pytest.fixture
def registered_stub(clean_registry):
    from dynamix.model.device import register_device
    register_device(_SlowStub())


def test_worker_resolves_off_thread_and_reports(qtbot, registered_stub):
    import numpy as np
    from dynamix.shell.worker import ResolveWorker

    layer = Layer(0, "l", "src0", Chain((DeviceRef("slow_stub", {}),)))
    w = ResolveWorker(layer, np.ones((4, 4)), Cache(), "src0")
    stages, results = [], []
    w.progress.connect(lambda s, f: stages.append((s, f)))   # test-side lambda is fine
    w.finished.connect(results.append)
    with qtbot.waitSignal(w.finished, timeout=5000):
        thread = w.start()
    thread.quit(); thread.wait()
    assert results[0].result["ok"] is True
    assert ("stub", 0.0) in stages and ("stub", 1.0) in stages
