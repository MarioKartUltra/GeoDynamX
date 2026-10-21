# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Saving a JSON document so that an interrupted save leaves the previous document in place."""
from __future__ import annotations

import contextlib
import json
import os
import uuid
from pathlib import Path


def write_json_atomically(path, payload) -> None:
    """Write ``json.dumps(payload, indent=2)`` to ``path`` as UTF-8 text, replacing the file whole.

    The text goes to a temporary file in the target's own directory, is flushed and fsynced, and
    ``os.replace`` then swaps it onto the target in one filesystem operation, atomic on macOS and
    on Windows since a sibling file always sits on the target's volume. A failure at any step
    removes the temporary file and leaves the previous document byte for byte. The temporary file
    is opened in text mode, so the bytes (line endings included) and default permissions are
    those of ``Path.write_text``. A symlinked target is resolved first, so the link survives and
    the file it names is the one replaced.
    """
    target = Path(os.path.realpath(path))
    text = json.dumps(payload, indent=2)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
