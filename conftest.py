# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Put src/ on sys.path so `import dynamix` works before the package is installed."""
import os
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

# Child processes do NOT inherit sys.path, only the environment. Several wtmm_backend
# tests assert the lazy-import property by shelling out to `sys.executable -c ...`, and
# under a src/ layout those subprocesses cannot find the package without this. EQSelect
# never needed it because its package sits at the repo root.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(SRC), os.environ.get("PYTHONPATH", "")) if p
)
