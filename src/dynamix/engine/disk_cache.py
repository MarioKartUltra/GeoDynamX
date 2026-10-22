# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Process-local, atomic pickle files for the result cache's disk tier."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pickle
import shutil
import threading


class DiskStore:
    def __init__(self, folder, max_bytes=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.max_bytes = (min(32_000_000_000, shutil.disk_usage(self.folder).free // 2)
                          if max_bytes is None else int(max_bytes))
        self._lock = threading.RLock()
        self._index = {}

    def record(self, key):
        with self._lock:
            return dict(self._index[key]) if key in self._index else None

    def records(self):
        with self._lock:
            return {k: dict(v) for k, v in self._index.items()}

    def write(self, key, value, version, metadata):
        stem = hashlib.sha256(key.encode()).hexdigest()
        path = self.folder / f"{stem}-{version}.pickle"
        temporary = path.with_suffix('.tmp')
        try:
            with temporary.open('wb') as stream:
                pickle.dump(value, stream, protocol=5)
            os.replace(temporary, path)
            record = {**metadata, 'path': path, 'bytes': path.stat().st_size, 'version': version}
        finally:
            temporary.unlink(missing_ok=True)
        removed = []
        with self._lock:
            previous = self._index.pop(key, None)
            self._index[key] = record
            if previous and previous['path'] != path:
                removed.append(previous)
            while sum(r['bytes'] for r in self._index.values()) > self.max_bytes:
                victim = next(iter(self._index))
                removed.append(self._index.pop(victim))
            kept = key in self._index
        for old in removed:
            self._unlink(old['path'])
        return kept

    def read(self, key):
        with self._lock:
            record = self._index.pop(key, None)
            if record is None:
                raise KeyError(key)
            self._index[key] = record
        with record['path'].open('rb') as stream:
            return pickle.load(stream), record

    def remove(self, key, version=None):
        with self._lock:
            record = self._index.get(key)
            if record is None or (version is not None and record['version'] != version):
                return False
            self._index.pop(key)
        self._unlink(record['path'])
        return True

    def clear(self):
        with self._lock:
            records, self._index = list(self._index.values()), {}
        for record in records:
            self._unlink(record['path'])

    @staticmethod
    def _unlink(path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A Windows reader can keep the file open; the next dead-process sweep removes it.
            pass

    def close(self):
        self.clear()
        try:
            self.folder.rmdir()
            if self.folder.parent.name == f"cache-{os.getpid()}":
                self.folder.parent.rmdir()
        except OSError:
            pass
