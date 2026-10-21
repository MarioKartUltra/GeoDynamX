# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Opt-in session diagnostics; expensive object census runs only at landings."""
from __future__ import annotations

from collections import Counter
from functools import wraps
import gc
import json
import os
import subprocess
import sys
import time


def enabled():
    return os.environ.get("DYNAMIX_TIMING") == "1"


def measure(name, *, report=False, event=None):
    """Time a bound method without changing its Qt receiver affinity."""
    def decorate(fn):
        @wraps(fn)
        def timed(self, *args, **kwargs):
            if not enabled():
                return fn(self, *args, **kwargs)
            if name == "scale_tick_ms" and (self._syncing or self.layer is None):
                return fn(self, *args, **kwargs)
            depth = getattr(self, "_timing_depth", 0)
            if report:
                self._timing_depth = depth + 1
            start = time.perf_counter()
            try:
                return fn(self, *args, **kwargs)
            finally:
                stats = self.__dict__.setdefault("_session_timing", {})
                stats[name] = stats.get(name, 0.0) + (time.perf_counter() - start) * 1000
                if report:
                    self._timing_depth = depth
                    if depth == 0:
                        emit(self, event or name)
        return timed
    return decorate


def fed(window, target):
    """Count distinct display targets actually updated between diagnostic lines."""
    if enabled():
        window.__dict__.setdefault("_timing_windows", set()).add(id(target))


def emit(window, event):
    """Paint counters cover completed paints since the preceding diagnostic line."""
    diagnostic_start = time.perf_counter()
    from dynamix.engine.cache import stage_nbytes
    stats = window.__dict__.setdefault("_session_timing", {})
    row = {"engine_ms": 0.0, "preview_ms": 0.0, "apply_ms": 0.0,
           "event": event, "pid": os.getpid(), "time": time.time(), **stats}
    stats.clear()
    layer = getattr(window, "layer", None)
    field = getattr(window, "field", None)
    row.update(layer_id=getattr(layer, "layer_id", None),
               layer_name=getattr(layer, "name", None),
               scale_idx=getattr(layer, "tags", {}).get("ui.scale_idx"),
               field_shape=getattr(getattr(field, "values", None), "shape", None),
               center_view=getattr(window, "_center_view", None))
    row["windows_fed"] = len(window.__dict__.get("_timing_windows", ()))
    window.__dict__.get("_timing_windows", set()).clear()
    canvas = getattr(window, "canvas", None)
    if canvas is not None:
        paints = canvas.__dict__.get("_session_timing", {})
        row.update({"paint_ms": paints.get("paint_ms", 0.0),
                    "paint_count": paints.get("paint_count", 0)})
        paints.clear()
    cache = window.cache
    entries = tuple(cache._store.values())
    row.update(cache_entries=len(entries),
               cache_bytes=sum(stage_nbytes(v) for v in entries),
               stage_entries=len(cache.stages))
    row["rss_bytes"] = _rss_bytes()
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        row.update(mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory())
    objects = gc.get_objects()
    row["python_objects"] = len(objects)
    if event == "landing_ms":
        row["object_types"] = Counter(type(o).__name__ for o in objects).most_common(10)
    from PySide6 import QtWidgets
    row["visible_windows"] = sum(w.isVisible() for w in QtWidgets.QApplication.topLevelWidgets())
    row["diagnostic_ms"] = (time.perf_counter() - diagnostic_start) * 1000
    print("DYNAMIX_TIMING " + json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)


def _rss_bytes():
    """Current resident memory, rather than the process high-water mark."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                        "nonpaged", "pagefile", "peak_pagefile")]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                                   ctypes.POINTER(Counters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                                           counters.cb):
                return counters.rss
            return None
        return int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
