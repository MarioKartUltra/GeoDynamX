#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
# Launch the DynamiX demo without installing the package.
#
#   scripts/demo.sh                    # the in-repo EBSD fixture (64x64, 70 um/px)
#   scripts/demo.sh path/to/field.npz  # any RasterField .npz
#
# Uses the xsmurf env if present, else whatever python3 is on PATH.
set -e
cd "$(dirname "$0")/.."

PY="$HOME/miniforge/envs/xsmurf/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

PYTHONPATH=src exec "$PY" -m dynamix.shell.demo "$@"
