# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A dropped shown result is restored by the worker on the next filter resolve."""
import threading
from PySide6 import QtWidgets

from tests.test_shell_window import loaded,stub_devices,window,_StubStack


def test_dropping_the_shown_analysis_never_computes_on_the_gui_thread(loaded,qtbot,monkeypatch):
    keys=loaded._cache_keys_for(loaded.layer)
    assert keys
    loaded.cache.invalidate(keys[-1])
    original=_StubStack.compute
    calls=[]
    def compute(self,*args,**kwargs):
        assert threading.current_thread() is not threading.main_thread()
        calls.append(1)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(_StubStack,'compute',compute)
    with qtbot.waitSignal(loaded.resolved,timeout=10000):
        loaded._reresolve()
    assert calls and loaded.cache.ready(keys[-1])
