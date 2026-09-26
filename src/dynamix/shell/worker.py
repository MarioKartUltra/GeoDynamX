# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ResolveWorker: runs ``dynamix.engine.resolve`` off the GUI thread.

The worker does no work of its own -- ``resolve`` is the engine's job, this class only gets it
off the thread that owns the widgets. Adapted from EQSelect's ``_WtmmWorker``
(1216-1255): a plain ``QObject`` moved to a bare ``QThread``,
never a ``QThread`` subclass, so it stays a small piece of Qt plumbing instead of accreting a
second identity as "the thread that resolves things."

**Consumers must connect every signal to a bound method, never a lambda or a free function.**
Qt decides whether a connection is queued (delivered on the receiver's own thread, via its event
loop) or direct (called immediately, on the emitting thread) by asking the receiver for its
thread affinity -- and only a bound method of a ``QObject`` has one. A lambda has no owning
``QObject``, so Qt cannot tell it lives on the GUI thread; the slot then runs *on this worker's
thread*, and touching a widget from there is a crash waiting to happen, not a guarantee. Bind the
slot to something Qt can place, or accept that it runs here.
"""
from __future__ import annotations

import threading

from PySide6 import QtCore

from dynamix.core.wtmm_backend import ComputeCancelled
from dynamix.engine import resolve
from dynamix.engine.resolve import preview_resolve


class ResolveWorker(QtCore.QObject):
    """Runs one ``resolve(layer, field, cache, source_id=...)`` on a dedicated ``QThread``.

    ``start()`` creates the thread, moves this worker onto it, and starts it running -- it does
    not tear the thread down. The caller keeps the returned ``QThread`` and is responsible for
    ``quit()``/``wait()`` once it is done with the result (the window chains finished/failed ->
    quit -> deleteLater).
    """

    progress = QtCore.Signal(str, float)
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)
    #: Progressive compute (design §3): distinct from ``error`` -- a cancelled run is
    #: not a failure, it is a superseded compute the window asked to abandon. The consumer tears
    #: the thread down and dispatches the next need, without an error notice.
    cancelled = QtCore.Signal()
    #: The finest-scale preview result, emitted BEFORE ``finished`` when ``preview=True`` -- the
    #: "show something now, don't wait for the whole stack" frame (design §3). Carries an ordinary
    #: result dict the canvas draws; the later ``finished`` replaces it with the full stack.
    partial = QtCore.Signal(object)

    def __init__(self, layer, field, cache, source_id, *, preview=False, prelude=()):
        super().__init__()
        #: ``[(ref_layer, ref_field, stamp)]`` a bus's LAYER sends need first, dependencies
        #: before dependents: each is resolved here, on this thread, and the raster it shows
        #: filed under its stamp (``core.bus.register_plane``) for the bus to read.
        self._prelude = list(prelude)
        self._layer = layer
        self._field = field
        self._cache = cache
        self._source_id = source_id
        self._preview = preview
        #: Set from the GUI thread via :meth:`cancel`; read cooperatively inside ``resolve`` at
        #: each WTMM stage boundary (``run_wtmm2d``'s ``cancel`` check). A ``threading.Event`` so
        #: the cross-thread read is safe without a lock.
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Ask the in-flight resolve to abandon at its next stage boundary. Safe to call from the
        GUI thread while ``_run`` executes on the worker thread."""
        self._cancel.set()

    def start(self) -> QtCore.QThread:
        thread = QtCore.QThread()
        self.moveToThread(thread)
        thread.started.connect(self._run)
        thread.start()
        return thread

    def _on_progress(self, stage: str, frac: float) -> None:
        """Bound method handed to ``resolve`` as its progress callback. Runs on this worker's
        thread; only re-emits, never touches a widget."""
        self.progress.emit(stage, frac)

    def _run(self) -> None:
        """The thread's entry point (connected to ``QThread.started``). Never lets an exception
        escape -- one always becomes ``error(str(exc))`` instead of an unhandled exception on a
        non-GUI thread, which Qt would otherwise just swallow silently."""
        try:
            if self._preview and not self._cancel.is_set():
                # Finest-scale-first (design §3): emit a cheap preview so the canvas shows the
                # finest scale's lines almost immediately, then keep going for the full stack.
                # A preview that isn't previewable (no leading wtmm2d) or that raises is simply
                # skipped -- never a reason to fail or delay the real compute.
                try:
                    prev = preview_resolve(self._layer, self._field)
                except Exception:
                    prev = None
                if prev is not None and not self._cancel.is_set():
                    self.partial.emit(prev)
            if self._prelude:
                from dynamix.core.bus import register_plane, shown_plane

                for ref_layer, ref_field, stamp in self._prelude:
                    ref = resolve(ref_layer, ref_field, self._cache,
                                  source_id=ref_layer.source_id, progress=self._on_progress,
                                  cancel=self._cancel.is_set)
                    try:
                        register_plane(stamp, shown_plane(ref.result, ref_field))
                    except ValueError as exc:
                        raise ValueError(f"{ref_layer.name}: {exc}") from exc
            renderable = resolve(
                self._layer, self._field, self._cache,
                source_id=self._source_id, progress=self._on_progress,
                cancel=self._cancel.is_set,
            )
        except ComputeCancelled:
            # Superseded by a newer edit -- not an error. The window drops it and dispatches the
            # current need; nothing partial was cached (run_wtmm2d checks BEFORE each stage).
            self.cancelled.emit()
            return
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(renderable)
